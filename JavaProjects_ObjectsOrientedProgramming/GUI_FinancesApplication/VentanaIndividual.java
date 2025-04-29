import javax.swing.*;
import java.awt.*;

public class VentanaIndividual extends JFrame {
    public VentanaIndividual(){
        super("Cálculo Individual");  //Ya no es necesario declararlo en main, ya que se hereda

        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); //cuando cierras la ventana, se termina el programa

        PanelResultados pd = new PanelResultados();
        this.add(pd);
        PanelControles pc = new PanelControles(pd);
        this.add(pc, BorderLayout.WEST);

        this.pack();                //Que se ajuste el tamaÃ±o de la ventana
        this.setVisible(true);      //Abra ventana y que sea visible
    }

    /*public static void main(String[] args) {
        VentanaIndividual individual = new VentanaIndividual();    //Instanciar

    }*/
}
