import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;


public class Copy {

	public static void copiarArchivo(String origen,String destino){
		try{
		
			BufferedReader bf=new BufferedReader(new FileReader(origen));
			PrintWriter pw=new PrintWriter(new FileWriter(destino)); 
			String linea;
			
			bf.readLine();
			
			while((linea=bf.readLine())!=null){
				
				pw.println(linea);
			}
			bf.close();
			pw.close();
		}catch(FileNotFoundException ex){
			System.out.println("No se encontró el archivo");
		}catch(IOException ex){
			System.out.println("Ocurrió un error con el archivo");
		}
	}
	
	public static void main(String[] args) {
		String origen="",
			   destino="";
		if(args.length!=0){
			origen=args[0];
			destino=args[1];
		}
		Copy.copiarArchivo(origen,destino);
	}

}
